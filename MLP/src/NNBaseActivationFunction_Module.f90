module mod_BaseActivationFunction
implicit none
    
!-------------------
! �����ࣺ����� |
!-------------------
type, abstract, public :: BaseActivationFunction

!||||||||||||    
contains   !|
!||||||||||||

    !* �����
    procedure(m_f), deferred, public :: f 
    !* �������������ļ����
    procedure(m_f_vect), deferred, public :: f_vect 
    !* ���������
    procedure(m_df), deferred, public :: df 
    !* �������������ļ��������
    procedure(m_df_vect), deferred, public :: df_vect  

end type BaseActivationFunction
!===================
    

!-------------------
! �����ࣺ�����ӿ� |
!-------------------	
abstract interface 

	!* ע������sigmod�ȼ���������ǵ�����������
	!* ������softmax������˵������������һ��������
	!* �Ƕ������������˺���Ӧ�������� x.

	!* �����
	subroutine m_f( this, index, x, y )
    use mod_Precision
    import :: BaseActivationFunction
	implicit none
		class(BaseActivationFunction), intent(inout) :: this
		!* index ��ʾ���� f( x(index) ) ��ֵ
		integer, intent(in) :: index
		real(PRECISION), dimension(:), intent(in) :: x
		real(PRECISION), intent(out) :: y
		
		
	end subroutine
	!====
    
    !* �������������ļ����
	subroutine m_f_vect( this, x, y )
    use mod_Precision
    import :: BaseActivationFunction
	implicit none
		class(BaseActivationFunction), intent(inout) :: this
		real(PRECISION), dimension(:), intent(in) :: x
		real(PRECISION), dimension(:), intent(out) :: y
		
	end subroutine
	!====
	
	!* �����һ�׵���
	subroutine m_df( this, index, x, dy  )
    use mod_Precision
    import :: BaseActivationFunction
	implicit none
		class(BaseActivationFunction), intent(inout) :: this
		!* index ��ʾ���� f'( x(index) ) ��ֵ
		integer, intent(in) :: index
		real(PRECISION), dimension(:), intent(in) :: x
		real(PRECISION), intent(out) :: dy

	end subroutine
	!====
	
	!* �������������ļ����һ�׵���
	subroutine m_df_vect( this, x, dy )
    use mod_Precision
    import :: BaseActivationFunction
	implicit none
		class(BaseActivationFunction), intent(inout) :: this
		real(PRECISION), dimension(:), intent(in) :: x
		real(PRECISION), dimension(:), intent(out) :: dy

	end subroutine
	!====

end interface
!===================
    
end module